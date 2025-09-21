import { createRootRoute, Link, Outlet } from '@tanstack/react-router'
import { Footer } from '../components/Footer'
import { KoFiButton } from '../components/KoFiButton'
import { SignedIn, SignedOut, SignInButton, SignUpButton, UserButton } from '@clerk/clerk-react'

function RootLayout() {
  return (
    <div className="min-h-screen w-full bg-gray-50 flex flex-col">
      <header className="w-full bg-white border-b">
        <div className="mx-auto max-w-[1280px] px-6 h-16 flex items-center justify-between">
          <div className="font-semibold">
            <Link to="/" className="hover:opacity-80">MangaFuse</Link>
          </div>
          <div className="flex items-center gap-3">
            <KoFiButton />
            <Link to="/pricing" className="text-gray-600 [&.active]:font-semibold">Pricing</Link>
            <Link to="/projects" className="text-gray-600 [&.active]:font-semibold">My Projects</Link>
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
      {/* Removed secondary sub-header to simplify navigation */}

      <main className="mx-auto max-w-[1280px] px-6 py-6 flex-1 w-full">
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}

export const Route = createRootRoute({
  component: RootLayout,
})