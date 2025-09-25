import { createFileRoute } from '@tanstack/react-router'
import { useAuth } from '@clerk/clerk-react'
import { useEffect, useState } from 'react'
import { createCheckoutSession, createPortalSession, fetchBillingStatus } from '../api'
import type { BillingStatus } from '../types'

function UsageBanner() {
	const { getToken, isSignedIn } = useAuth()
	const [status, setStatus] = useState<BillingStatus | null>(null)
	const [error, setError] = useState<string | null>(null)

	useEffect(() => {
		if (!isSignedIn) return;

		async function loadStatus() {
			try {
				const data = await fetchBillingStatus(getToken)
				setStatus(data)
			} catch (err) {
				setError(err instanceof Error ? err.message : 'Could not load usage status')
			}
		}
		loadStatus()
	}, [getToken, isSignedIn])

	// Helper to calculate and format the reset date
	const getResetDate = (): string | null => {
		if (!status) return null;

		// Use period_end when available (anniversary-based for all plans)
		if (status.period_end) {
			const date = new Date(status.period_end);
			return date.toLocaleDateString(undefined, { month: 'long', day: 'numeric' });
		}

		// Fallback: end of current calendar month
		const now = new Date();
		const endOfMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0);
		return endOfMonth.toLocaleDateString(undefined, { month: 'long', day: 'numeric' });
	}

	// Don't render anything for signed-out users
	if (!isSignedIn) {
		return null;
	}

	if (error) {
		return <div className="p-3 mb-4 text-sm bg-red-50 text-red-700 rounded border border-red-200">{error}</div>
	}

	if (!status) {
		return <div className="p-3 mb-4 text-sm bg-gray-100 rounded border animate-pulse">Loading usage...</div>
	}

	const planName = status.plan_id === 'free' ? 'Free Plan' : 'Pro Plan'
	const resetDate = getResetDate();

	return (
		<div className="p-3 mb-4 text-sm bg-blue-50 text-blue-800 rounded border border-blue-200">
			<p>You are on the <strong>{planName}</strong> plan. You have used {status.project_count} of your {status.project_limit}{' '}
			monthly projects.</p>
			{resetDate && <p className="mt-1">Your limit resets on <strong>{resetDate}</strong>.</p>}
		</div>
	)
}

function PricingPage() {
  const { getToken } = useAuth()
  const [loading, setLoading] = useState('') // To track which button is loading

  const handleSubscribe = async () => {
    setLoading('subscribe')
    try {
      const { url } = await createCheckoutSession(getToken)
      window.location.href = url
    } catch (err) {
      alert(err instanceof Error ? err.message : 'An error occurred.')
      setLoading('')
    }
  }

  const handleManageSubscription = async () => {
    setLoading('manage')
    try {
      const { url } = await createPortalSession(getToken)
      window.location.href = url
    } catch (err) {
      alert(err instanceof Error ? err.message : 'An error occurred.')
      setLoading('')
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
			<UsageBanner />
      <h1 className="text-3xl font-bold text-center">Pricing Plans</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-lg border">
          <h2 className="text-xl font-semibold">Free</h2>
          <p className="text-gray-600 mt-2">Get started for free</p>
          <div className="text-3xl font-bold my-4">$0 <span className="text-lg font-normal">/ month</span></div>
          <ul className="space-y-2 text-sm text-gray-700">
            <li>✅ 5 projects / month</li>
            <li>✅ Community access</li>
            <li>✅ Basic support</li>
          </ul>
        </div>

        <div className="bg-white p-6 rounded-lg border-2 border-blue-600">
          <h2 className="text-xl font-semibold">Pro</h2>
          <p className="text-blue-600 mt-2">For serious creators</p>
          <div className="text-3xl font-bold my-4">$10 <span className="text-lg font-normal">/ month</span></div>
          <ul className="space-y-2 text-sm text-gray-700">
            <li>✅ 200 projects / month</li>
            <li>✅ Feature requests</li>
            <li>✅ Priority support</li>
          </ul>
          <button
            onClick={handleSubscribe}
            disabled={!!loading}
            className="mt-6 w-full py-2 px-4 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {loading === 'subscribe' ? 'Redirecting...' : 'Upgrade to Pro'}
          </button>
        </div>

        <div className="bg-white p-6 rounded-lg border">
          <h2 className="text-xl font-semibold">Enterprise</h2>
          <p className="text-gray-600 mt-2">For creators with custom needs</p>
          <div className="text-3xl font-bold my-4">Custom</div>
          <ul className="space-y-2 text-sm text-gray-700">
            <li>✅ More projects / month</li>
            <li>✅ Custom integrations</li>
            <li>✅ Dedicated support</li>
          </ul>
          <a
            href="mailto:a.liu27568@gmail.com"
            className="mt-6 w-full block text-center py-2 px-4 rounded bg-gray-800 text-white hover:bg-gray-900"
          >
            Contact Us
          </a>
        </div>
      </div>
      <div className="text-center mt-4">
        <button
          onClick={handleManageSubscription}
          disabled={!!loading}
          className="text-sm text-blue-600 hover:underline disabled:opacity-50"
        >
          {loading === 'manage' ? 'Redirecting...' : 'Manage your existing subscription'}
        </button>
      </div>
    </div>
  )
}

export const Route = createFileRoute('/pricing')({
  component: PricingPage,
})