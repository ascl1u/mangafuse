import { EXTERNAL_URLS } from '../constants'

export function KoFiButton() {
  return (
    <a
      href={EXTERNAL_URLS.kofi}
      target="_blank"
      rel="noopener noreferrer"
      aria-label="Support me on Ko-fi"
      title="Support me on Ko-fi"
      className="inline-flex items-center"
    >
      <img src="/icons/support_me_on_kofi_dark.png" alt="Support me on Ko-fi" className="h-6 w-auto" />
    </a>
  )
}