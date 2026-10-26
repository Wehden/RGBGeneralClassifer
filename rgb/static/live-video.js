/* Decode complete MJPEG parts before replacing the visible image. */
class LiveVideo {
  constructor(image, report, {timeout = 15000, retry = 5000} = {}) {
    Object.assign(this, {image, report, timeout, retry});
    this.generation = 0;
    this.url = null;
    this.objectURL = null;
  }

  start(url) {
    const changed = this.url !== url;
    // A different camera must not inherit the previous camera's picture.
    this.stop(!changed);
    this.url = url;
    this.connect(this.generation);
  }

  stop(preserveFrame = false) {
    this.generation++;
    clearTimeout(this.retryTimer);
    clearTimeout(this.watchdog);
    this.controller?.abort();
    this.url = null;
    if (!preserveFrame && this.objectURL) {
      this.image.removeAttribute('src');
      URL.revokeObjectURL(this.objectURL);
      this.objectURL = null;
    }
  }

  stale() {
    this.report(this.objectURL ? 'Stale · showing last frame · reconnecting' : 'Camera unavailable · reconnecting');
  }

  async connect(generation) {
    const valid = () => generation === this.generation;
    const controller = this.controller = new AbortController();
    const arm = () => {
      clearTimeout(this.watchdog);
      this.watchdog = setTimeout(() => {
        if (valid()) this.stale();
        controller.abort();
      }, this.timeout);
    };
    this.report(this.objectURL ? 'Reconnecting · showing last frame' : 'Connecting live frame…');
    arm();
    try {
      const response = await fetch(this.url, {signal: controller.signal, cache: 'no-store'});
      if (!response.ok) throw new Error('Stream HTTP ' + response.status);
      const match = /boundary\s*=\s*(?:"([^"]+)"|([^;\s]+))/i.exec(response.headers.get('Content-Type') || '');
      const name = match && (match[1] || match[2]);
      if (!name || name.length > 200) throw new Error('Invalid MJPEG boundary');
      const boundary = new TextEncoder().encode('--' + name);
      const delimiter = new TextEncoder().encode('\r\n--' + name);
      const separator = new Uint8Array([13, 10, 13, 10]);
      const limit = 4 * 1024 * 1024;
      const find = (data, needle, start = 0) => {
        outer: for (let i = start; i <= data.length - needle.length; i++) {
          for (let j = 0; j < needle.length; j++) if (data[i + j] !== needle[j]) continue outer;
          return i;
        }
        return -1;
      };
      let buffer = new Uint8Array();
      const reader = response.body.getReader();
      try {
        while (valid()) {
          const {value, done} = await reader.read();
          if (done) throw new Error('Stream ended');
          if (buffer.length + value.length > limit) throw new Error('Oversized MJPEG part');
          const joined = new Uint8Array(buffer.length + value.length);
          joined.set(buffer);
          joined.set(value, buffer.length);
          buffer = joined;
          while (valid()) {
            const first = find(buffer, boundary);
            if (first < 0) break;
            const headersEnd = find(buffer, separator, first + boundary.length);
            if (headersEnd < 0) break;
            const headers = new TextDecoder().decode(buffer.subarray(first + boundary.length, headersEnd));
            const length = /(?:^|\r\n)Content-Length:\s*(\d+)\s*(?:\r\n|$)/i.exec(headers);
            const start = headersEnd + separator.length;
            // Length-bearing parts can be displayed without waiting for another frame.
            const end = length ? start + Number(length[1]) : find(buffer, delimiter, start);
            if (length && end - start > limit) throw new Error('Oversized JPEG');
            if (end < 0 || end > buffer.length) break;
            const objectURL = URL.createObjectURL(new Blob([buffer.slice(start, end)], {type: 'image/jpeg'}));
            try {
              const candidate = new Image();
              candidate.src = objectURL;
              let abortDecode;
              try {
                if (controller.signal.aborted) throw new Error('Stream aborted');
                await Promise.race([
                  candidate.decode(),
                  new Promise((resolve, reject) => {
                    abortDecode = () => reject(new Error('Decode aborted'));
                    controller.signal.addEventListener('abort', abortDecode, {once: true});
                  }),
                ]);
              } finally {
                if (abortDecode) controller.signal.removeEventListener('abort', abortDecode);
              }
              if (!valid() || controller.signal.aborted) {
                URL.revokeObjectURL(objectURL);
                return;
              }
              const old = this.objectURL;
              this.image.src = objectURL;
              this.objectURL = objectURL;
              if (old) URL.revokeObjectURL(old);
              this.report('LIVE');
              arm();
            } catch (error) {
              URL.revokeObjectURL(objectURL);
              throw error;
            }
            buffer = buffer.slice(end);
          }
        }
      } finally {
        await reader.cancel().catch(() => {});
        reader.releaseLock();
      }
    } catch (error) {
      if (valid()) this.stale();
    } finally {
      if (valid()) {
        clearTimeout(this.watchdog);
        this.retryTimer = setTimeout(() => this.connect(generation), this.retry);
      }
    }
  }
}
window.LiveVideo = LiveVideo;
