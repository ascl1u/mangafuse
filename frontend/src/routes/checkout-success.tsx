import { createFileRoute } from '@tanstack/react-router'
import { useAuth } from '@clerk/clerk-react'
import { useEffect, useState } from 'react'
import { syncSubscription } from '../api'

function CheckoutSuccessPage() {
  const { getToken } = useAuth()
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    async function performSync() {
      try {
        await syncSubscription(getToken)
        if (isMounted) {
          window.location.href = '/projects'
        }
      } catch (err) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'An unexpected error occurred.')
        }
      }
    }

    performSync()

    return () => { isMounted = false }
  }, [getToken])

  return (
    <div className="max-w-2xl mx-auto text-center py-10">
      {error ? (
        <>
          <h1 className="text-2xl font-semibold text-red-600">Sync Failed</h1>
          <p className="mt-2 text-gray-600">There was an issue confirming your subscription.</p>
          <p className="mt-1 text-sm text-red-700 bg-red-50 p-3 rounded">{error}</p>
        </>
      ) : (
        <>
          <h1 className="text-2xl font-semibold">Finalizing your subscription...</h1>
          <p className="mt-2 text-gray-600">Please wait while we confirm your payment details.</p>
        </>
      )}
    </div>
  )
}

export const Route = createFileRoute('/checkout-success')({
  component: CheckoutSuccessPage,
})
