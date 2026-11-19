from __future__ import annotations

import codecs


class SSEParser:
    """Incrementally parse an SSE byte stream into data payloads."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self._data_lines: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        self._buffer += self._decoder.decode(chunk)
        return self._drain(final=False)

    def close(self) -> list[str]:
        self._buffer += self._decoder.decode(b"", final=True)
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[str]:
        events: list[str] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if line == "":
                if self._data_lines:
                    events.append("\n".join(self._data_lines))
                    self._data_lines = []
            elif line.startswith(":"):
                continue
            elif line.startswith("data:"):
                self._data_lines.append(line[5:].removeprefix(" "))
        if final and self._buffer:
            line = self._buffer.removesuffix("\r")
            if line.startswith("data:"):
                self._data_lines.append(line[5:].removeprefix(" "))
            self._buffer = ""
        if final and self._data_lines:
            events.append("\n".join(self._data_lines))
            self._data_lines = []
        return events
