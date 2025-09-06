import { createFileRoute } from '@tanstack/react-router'
import { useAuth } from '@clerk/clerk-react'
import { useState } from 'react'
import { createCheckoutSession, createPortalSession } from '../api'

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
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold text-center">Pricing Plans</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg border">
          <h2 className="text-xl font-semibold">Free</h2>
          <p className="text-gray-600 mt-2">Get started for free</p>
          <div className="text-3xl font-bold my-4">$0 <span className="text-lg font-normal">/ month</span></div>
          <ul className="space-y-2 text-sm text-gray-700">
            <li>✅ 5 Projects / month</li>
            <li>✅ Basic AI models</li>
            <li>✅ Community support</li>
          </ul>
        </div>

        <div className="bg-white p-6 rounded-lg border-2 border-blue-600">
          <h2 className="text-xl font-semibold">Pro</h2>
          <p className="text-blue-600 mt-2">For serious creators</p>
          <div className="text-3xl font-bold my-4">$10 <span className="text-lg font-normal">/ month</span></div>
          <ul className="space-y-2 text-sm text-gray-700">
            <li>✅ 300 Projects / month</li>
            <li>✅ Advanced AI models</li>
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