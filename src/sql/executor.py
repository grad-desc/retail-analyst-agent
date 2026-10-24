"""SafeBigQueryRunner: extends the client-provided BigQueryRunner (src/bq_client.py, unchanged)
with dry-run cost guard, byte cap, timeout, row cap, transient-error retry and error classes
that map 1:1 to the agent's recovery strategies.
"""
import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from google.api_core import exceptions as gexc
from google.cloud import bigquery
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential_jitter)

from src.bq_client import BigQueryRunner

from .errors import BQUnavailable, QueryTooExpensive, SQLSyntaxError  # noqa: F401 (re-exported)
from .schema import SAFE_COLUMNS

log = logging.getLogger(__name__)

TRANSIENT = (gexc.ServiceUnavailable, gexc.InternalServerError,
             gexc.TooManyRequests, gexc.DeadlineExceeded)


class SafeBigQueryRunner(BigQueryRunner):
    def __init__(self, *args, max_bytes: int = 1 * 1024**3, max_rows: int = 10_000,
                 timeout_s: int = 60, location: str = "US", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_bytes, self.max_rows = max_bytes, max_rows
        self.timeout_s, self.location = timeout_s, location

    def dry_run(self, sql: str) -> int:
        """Free: catches syntax/semantic errors and estimates cost before real execution."""
        cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            job = self.client.query(sql, job_config=cfg, location=self.location)
        except gexc.BadRequest as e:
            raise SQLSyntaxError(self._clean(e)) from e
        except TRANSIENT as e:
            raise BQUnavailable(str(e)) from e
        if job.total_bytes_processed > self.max_bytes:
            raise QueryTooExpensive(
                f"Query would process {job.total_bytes_processed / 1e9:.1f} GB, limit is "
                f"{self.max_bytes / 1e9:.1f} GB. Add date filters or select fewer columns.")
        return job.total_bytes_processed

    def execute_query(self, sql_query: str, trace_id: Optional[str] = None) -> pd.DataFrame:
        self.dry_run(sql_query)
        started = time.perf_counter()
        try:
            df, job = self._run(sql_query, trace_id)
        except gexc.BadRequest as e:
            raise SQLSyntaxError(self._clean(e)) from e
        except gexc.Forbidden as e:  # incl. maximum_bytes_billed exceeded
            raise QueryTooExpensive(self._clean(e)) from e
        except TRANSIENT as e:
            raise BQUnavailable(str(e)) from e
        truncated = len(df) > self.max_rows
        df = df.head(self.max_rows)
        df.attrs.update(job_id=job.job_id, truncated=truncated, bytes_billed=job.total_bytes_billed)
        log.info("bq_query_done", extra={
            "trace_id": trace_id, "job_id": job.job_id, "rows": len(df),
            "bytes_billed": job.total_bytes_billed, "truncated": truncated,
            "latency_ms": int((time.perf_counter() - started) * 1000)})
        return df

    @retry(retry=retry_if_exception_type(TRANSIENT), stop=stop_after_attempt(3),
           wait=wait_exponential_jitter(initial=1, max=10), reraise=True)
    def _run(self, sql: str, trace_id: Optional[str]):
        cfg = bigquery.QueryJobConfig(
            maximum_bytes_billed=self.max_bytes,
            labels={"app": "retail-agent", "trace_id": (trace_id or "na").replace("-", "")[:60].lower()})
        job = self.client.query(sql, job_config=cfg, location=self.location)
        rows = job.result(timeout=self.timeout_s, max_results=self.max_rows + 1)
        return rows.to_dataframe(), job

    def get_safe_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Allowlist-filtered schema: the only schema the LLM and the user ever see."""
        if table_name not in SAFE_COLUMNS:
            raise ValueError(f"Table '{table_name}' is not available.")
        allowed = set(SAFE_COLUMNS[table_name])
        return [c for c in self.get_table_schema(table_name) if c["name"] in allowed]

    @staticmethod
    def _clean(e: Exception) -> str:
        return getattr(e, "message", str(e)).split("\n")[0][:500]
