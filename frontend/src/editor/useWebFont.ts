const cache = new Map<string, Promise<void>>()

export function loadWebFontOnce(family: string, url: string): Promise<void> {
  const key = `${family}|${url}`
  const existing = cache.get(key)
  if (existing) return existing
  if (document.fonts && [...document.fonts].some((f) => f.family === family)) {
    const p = Promise.resolve()
    cache.set(key, p)
    return p
  }
  const p = (async () => {
    try {
      const font = new FontFace(family, `url(${url})`)
      await font.load()
      document.fonts.add(font)
    } catch {
      // swallow; fallback fonts will be used
    }
  })()
  cache.set(key, p)
  return p
}


