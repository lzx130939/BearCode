"""Keep failed stream attempts out of committed answers and learning data."""
from __future__ import annotations

from .ui import print_info


class StreamAttempts:
    def _capture_stream_output(self):
        """捕获当前对话和总的输出缓冲区的长度"""
        return {"turn": len(self._turn_output_buffer or []),
                "output": len(self._output_buffer or [])}

    def _discard_stream_output(self, state) -> None:
        """丢弃自state以后的所有流式输出"""
        emitted = self._turn_output_buffer is not None and len(self._turn_output_buffer) > state["turn"]
        if self._turn_output_buffer is not None:
            del self._turn_output_buffer[state["turn"]:]
        if self._output_buffer is not None:
            del self._output_buffer[state["output"]:]
        elif emitted:
            # Terminal text cannot be recalled; explicitly invalidate the partial
            # attempt. Browser adapters replace their provisional entries.
            print_info("Response interrupted; the partial answer above was discarded.")

    async def _with_stream_retry(self, request, *, retry):
        async def attempt():
            # 在执行_do前先统计好每个buffer的长度，等_do执行完后，如果出现异常，就把buffer恢复到之前的状态
            state = self._capture_stream_output()
            try:
                return await request()
            except BaseException:
                self._discard_stream_output(state)
                raise
        return await retry(attempt)
