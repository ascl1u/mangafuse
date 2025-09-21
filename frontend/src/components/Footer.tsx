import { EXTERNAL_URLS } from '../constants'

export function Footer() {
  const year = new Date().getFullYear()
  return (
    <footer className="w-full bg-white border-t">
      <nav aria-label="Footer" className="mx-auto max-w-[1280px] px-6 py-6 flex items-center justify-between text-sm text-gray-600">
        <span>© {year} MangaFuse</span>
        <div className="flex items-center gap-4">
          <a
            href={EXTERNAL_URLS.x}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="X"
            className="text-gray-600"
          >
            <img src="/icons/x.svg" alt="" aria-hidden="true" className="h-5 w-5" />
          </a>
          <a
            href={EXTERNAL_URLS.discord}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Discord"
            className="text-gray-600"
          >
            <img src="/icons/discord.svg" alt="" aria-hidden="true" className="h-5 w-5" />
          </a>
          <a
            href={EXTERNAL_URLS.tos}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-white rounded"
          >
            Terms of Service
          </a>
        </div>
      </nav>
    </footer>
  )
}


