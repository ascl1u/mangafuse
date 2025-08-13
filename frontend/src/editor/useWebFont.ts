let loadingPromise: Promise<void> | null = null

export function loadWebFontOnce(family: string, url: string): Promise<void> {
  if (loadingPromise) return loadingPromise
  if (document.fonts && [...document.fonts].some((f) => f.family === family)) {
    loadingPromise = Promise.resolve()
    return loadingPromise
  }
  loadingPromise = (async () => {
    try {
      const font = new FontFace(family, `url(${url})`)
      await font.load()
      document.fonts.add(font)
    } catch {
      // swallow; fallback fonts will be used
    }
  })()
  return loadingPromise
}


