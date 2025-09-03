import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { ClerkProvider, useAuth } from '@clerk/clerk-react'
import { RouterProvider, createRouter } from '@tanstack/react-router'

// Import the generated route tree
import { routeTree } from './routeTree.gen'

const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY as string | undefined
if (!PUBLISHABLE_KEY) {
  throw new Error('Missing VITE_CLERK_PUBLISHABLE_KEY')
}

// Create a new router instance with minimal context
export type RouterContext = {
  getToken: () => Promise<string | null>
}

const router = createRouter({
  routeTree,
  context: {
    // Placeholder; actual getToken is provided at render via RouterProvider
    getToken: async () => null,
  } as RouterContext,
})

// Register the router instance for type safety
declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

export function AppRouterProvider() {
  const { getToken } = useAuth()
  return <RouterProvider router={router} context={{ getToken }} />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ClerkProvider publishableKey={PUBLISHABLE_KEY} afterSignOutUrl="/">
      <AppRouterProvider />
    </ClerkProvider>
  </StrictMode>,
)