export function Footer() {
  const year = new Date().getFullYear()
  return (
    <footer className="w-full bg-white border-t">
      <nav aria-label="Footer" className="mx-auto max-w-[1280px] px-6 py-6 flex items-center justify-between text-sm text-gray-600">
        <span>© {year} MangaFuse</span>
        <a
          href="/tos.html"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-white rounded"
        >
          Terms of Service
        </a>
      </nav>
    </footer>
  )
}


