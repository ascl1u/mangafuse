import { createRootRoute, Link, Outlet } from '@tanstack/react-router'
import { SignedIn, SignedOut, SignInButton, SignUpButton, UserButton } from '@clerk/clerk-react'

function RootLayout() {
  const X_URL = import.meta.env.VITE_X_URL || '#'
  const DISCORD_URL = import.meta.env.VITE_DISCORD_URL || '#'
  return (
    <div className="min-h-screen w-full bg-gray-50">
      <header className="w-full bg-white border-b">
        <div className="mx-auto max-w-[1280px] px-6 h-16 flex items-center justify-between">
          <div className="font-semibold">
            <Link to="/" className="hover:opacity-80">MangaFuse</Link>
          </div>
          <div className="flex items-center gap-3">
            <a href={X_URL} target="_blank" rel="noopener noreferrer" aria-label="X" className="text-gray-600">
              <img src="/icons/x.svg" alt="" aria-hidden="true" className="h-5 w-5" />
            </a>
            <a href={DISCORD_URL} target="_blank" rel="noopener noreferrer" aria-label="Discord" className="text-gray-600">
              <img src="/icons/discord.svg" alt="" aria-hidden="true" className="h-5 w-5" />
            </a>
            <Link to="/pricing" className="text-gray-600 [&.active]:font-semibold">Pricing</Link>
            <SignedOut>
              <SignInButton />
              <SignUpButton />
            </SignedOut>
            <SignedIn>
              <UserButton />
            </SignedIn>
          </div>
        </div>
      </header>
      <div className="w-full bg-white border-b">
        <div className="mx-auto max-w-[1280px] px-6 h-10 flex items-center text-gray-500 gap-3">
          <div className="px-3 py-1 rounded border bg-gray-100 opacity-60 cursor-not-allowed">Current job</div>
          <button onClick={() => window.location.reload()} className="px-3 py-1.5 rounded bg-blue-600 text-white">New job</button>
        </div>
      </div>

      <main className="mx-auto max-w-[1280px] px-6 py-6">
        <Outlet />
      </main>
    </div>
  )
}

export const Route = createRootRoute({
  component: RootLayout,
})