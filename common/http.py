"""Keep accepted live request and stream buffers in RAM, including slow clients."""
OPTIONS={'max_request_body_size':2_000_000,'inbuf_overflow':2_000_001,
         'outbuf_high_watermark':1_000_000,'outbuf_overflow':4_000_000}
