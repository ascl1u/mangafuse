import { useMemo, useState } from 'react'
import { SignUpButton } from '@clerk/clerk-react'
import type { EditorPayload } from '../types'
import { EditorCanvas } from '../editor/EditorCanvas'
import { EXTERNAL_URLS } from '../constants'

export function GuestDemo() {
  // State to track the current stage of the demo: 'raw', 'cleaned', or 'final'
  const [demoState, setDemoState] = useState<'raw' | 'cleaned' | 'final'>('raw')

  // The core of the reuse strategy: create a "synthetic" payload for EditorCanvas
  // based on the current demo stage.
  const syntheticEditorPayload = useMemo<EditorPayload>(() => {
    const base: EditorPayload = { image_url: '/demo/raw.jpg', width: 0, height: 0, bubbles: [] }
    if (demoState === 'raw') return { ...base, image_url: '/demo/raw.jpg', text_layer_url: undefined, bubbles: [] }
    if (demoState === 'cleaned') return { ...base, image_url: '/demo/cleaned.png', text_layer_url: undefined, bubbles: [] }
    return { ...base, image_url: '/demo/cleaned.png', text_layer_url: '/demo/text_layer.png', bubbles: [] }
  }, [demoState])

  // Determine the button text and the next state to transition to
  const buttonProps = useMemo(() => {
    if (demoState === 'raw') {
      return { text: 'Clean Text', nextState: 'cleaned' as const }
    }
    if (demoState === 'cleaned') {
      return { text: 'Add Translation', nextState: 'final' as const }
    }
    return null // No button in the final state
  }, [demoState])

  // Demo is non-interactive; no store usage needed

  return (
    <div className="space-y-4">
      <div className="p-4 text-center bg-blue-50 text-blue-800 rounded border border-blue-200">
        <h2 className="font-bold">Welcome to MangaFuse!</h2>
        <p className="text-sm">
          Follow the steps to see how it works, then sign up to try your own images.{' '}
          If you would like to make contributions, check out the{' '}
          <a
            href={EXTERNAL_URLS.github}
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-blue-900"
          >
            GitHub repo
          </a>
          .
        </p>
      </div>

      <div className="grid gap-6" style={{ gridTemplateColumns: '320px 1fr', columnGap: 24 }}>
        <div className="space-y-4">
          <div className="bg-white p-4 rounded border">
            <div className="font-medium mb-2">Controls</div>
            {buttonProps ? (
              <button
                onClick={() => setDemoState(buttonProps.nextState)}
                className="w-full px-4 py-2 rounded bg-blue-600 text-white font-semibold hover:bg-blue-700 transition-all"
              >
                {buttonProps.text}
              </button>
            ) : (
              <div className="text-center p-4 bg-green-50 text-green-800 rounded border border-green-200">
                <p className="font-semibold">Complete!</p>
                <SignUpButton>
                  <button className="w-full mt-2 px-4 py-2 rounded bg-green-600 text-white font-semibold hover:bg-green-700">
                    Sign Up to Start Your Project
                  </button>
                </SignUpButton>
              </div>
            )}
          </div>
        </div>

        <div className="bg-white p-2 rounded border min-w-0">
          <EditorCanvas
            editor={syntheticEditorPayload}
            onSelect={() => {}}
            pendingEditIds={[]}
            revision={0}
          />
        </div>
      </div>
    </div>
  )
}