import { createFileRoute } from '@tanstack/react-router'

function PricingPage() {
  return (
    <div className="max-w-2xl">
      <h1 className="text-xl font-semibold mb-3">Pricing</h1>
      <p className="text-sm text-gray-700">Coming soon.</p>
    </div>
  )
}

export const Route = createFileRoute('/pricing')({
  component: PricingPage,
})