from llmbench.sse import SSEParser


def test_parses_fragmented_utf8_and_crlf() -> None:
    parser = SSEParser()
    raw = (
        b'data: {"choices":[{"delta":{"content":"\xe4\xbd\xa0"}}]}\r\n\r\n'
        b"data: [DONE]\r\n\r\n"
    )
    events: list[str] = []
    for index in range(len(raw)):
        events.extend(parser.feed(raw[index : index + 1]))
    assert events == ['{"choices":[{"delta":{"content":"你"}}]}', "[DONE]"]


def test_parses_multiline_data_and_ignores_comments() -> None:
    parser = SSEParser()
    events = parser.feed(b": keepalive\n\ndata: first\ndata: second\n\n")
    assert events == ["first\nsecond"]


def test_close_flushes_event_without_trailing_blank_line() -> None:
    parser = SSEParser()
    assert parser.feed(b"data: tail") == []
    assert parser.close() == ["tail"]


def test_ignores_non_data_fields() -> None:
    parser = SSEParser()
    assert parser.feed(b"event: message\nid: 3\nretry: 1000\n\ndata: x\n\n") == ["x"]
