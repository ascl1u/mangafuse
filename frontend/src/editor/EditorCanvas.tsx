import { useEffect, useMemo, useRef, useState } from 'react'
import { Stage, Layer, Image as KonvaImage, Group, Rect, Circle } from 'react-konva'
import type { EditorPayload, EditorBubble } from '../types'

type Props = {
  editor: EditorPayload
  selectedId?: number
  onSelect: (id: number) => void
  pendingEditIds: number[]
  disabled?: boolean
  hasEditFailed?: boolean
  revision: number
}

function useHtmlImage(src?: string): HTMLImageElement | undefined {
  const [img, setImg] = useState<HTMLImageElement>()
  useEffect(() => {
    if (!src) {
      setImg(undefined)
      return
    }
    const image = new Image()
    image.crossOrigin = 'anonymous'
    image.onload = () => setImg(image)
    image.src = src
    return () => {
      setImg(undefined)
    }
  }, [src])
  return img
}

function polygonBBox(poly: [number, number][]) {
  const xs = poly.map((p) => p[0])
  const ys = poly.map((p) => p[1])
  const x0 = Math.floor(Math.min(...xs))
  const y0 = Math.floor(Math.min(...ys))
  const x1 = Math.ceil(Math.max(...xs))
  const y1 = Math.ceil(Math.max(...ys))
  return { x0, y0, x1, y1, w: Math.max(1, x1 - x0), h: Math.max(1, y1 - y0) }
}

// Helper function to determine error type from error message
function getErrorType(errorMessage: string): 'translation' | 'typeset' {
  return errorMessage === "Translation failed" ? 'translation' : 'typeset'
}

// Helper function to get error styling based on type
function getErrorStyle(errorType: 'translation' | 'typeset'): { strokeColor: string } {
  return {
    translation: { strokeColor: "#eab308" }, // Yellow for translation errors
    typeset: { strokeColor: "#ef4444" }      // Red for typeset errors
  }[errorType]
}

export function EditorCanvas({ editor, selectedId, onSelect, pendingEditIds, disabled, revision }: Props) {
  const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
  const imageUrl = `${API_BASE}${editor.image_url}`
  const img = useHtmlImage(imageUrl)

  // ✅ Use the stable `revision` prop for cache-busting
  const textLayerUrl = editor.text_layer_url
    ? `${API_BASE}${editor.text_layer_url}?rev=${revision}`
    : undefined
  const textLayerImg = useHtmlImage(textLayerUrl)

  // Fit-to-width sizing: container width is measured via ref
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState<number>(800)
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(() => {
      setContainerWidth(el.clientWidth)
    })
    obs.observe(el)
    setContainerWidth(el.clientWidth)
    return () => obs.disconnect()
  }, [])

  // Robust sizing: fall back to intrinsic image size if editor payload is missing dims
  const baseWidth = (typeof editor.width === 'number' && editor.width > 0)
    ? editor.width
    : (img?.naturalWidth ?? 1)
  const baseHeight = (typeof editor.height === 'number' && editor.height > 0)
    ? editor.height
    : (img?.naturalHeight ?? 1)
  const aspect = baseHeight / baseWidth
  const stageWidth = Math.max(100, containerWidth)
  const stageHeight = Math.max(100, Math.round(stageWidth * aspect))

  // Scale from image coords to stage coords
  const scale = stageWidth / baseWidth

  const bubbles = editor.bubbles as EditorBubble[]

  const polygons = useMemo(() => {
    return bubbles.map((b) => b.polygon.map(([x, y]) => [x * scale, y * scale]) as [number, number][])
  }, [bubbles, scale])

  const selectedIdx = bubbles.findIndex((b) => b.id === selectedId)

  return (
    <div ref={containerRef} className="w-full">
      <Stage width={stageWidth} height={stageHeight}>
        <Layer listening={false}>
          {img && <KonvaImage image={img} width={stageWidth} height={stageHeight} />}
        </Layer>
        <Layer listening={false}>
          {/* Show text layer only when not updating to avoid flashing stale text */}
          {textLayerImg && !disabled && (
            <KonvaImage image={textLayerImg} width={stageWidth} height={stageHeight} />
          )}
        </Layer>
        {disabled && (
          <Layer listening={false}>
            {/* Subtle overlay to indicate updating state */}
            <Rect x={0} y={0} width={stageWidth} height={stageHeight} fill="rgba(255,255,255,0.3)" />
          </Layer>
        )}
        <Layer>
          {polygons.map((poly, idx) => {
            const b = bubbles[idx]
            const isSelected = idx === selectedIdx
            const rect = polygonBBox(poly)
            const centerX = rect.x0 + rect.w / 2
            const centerY = rect.y0 + rect.h / 2
            const spinnerRadius = Math.max(8, Math.min(20, Math.min(rect.w, rect.h) * 0.1))
            const isPending = pendingEditIds.includes(b.id)
            return (
              <Group key={b.id} onClick={() => !disabled && onSelect(b.id)}>
                {/* invisible hit rect for selection */}
                <Rect
                  x={rect.x0}
                  y={rect.y0}
                  width={rect.w}
                  height={rect.h}
                  fill="rgba(0,0,0,0.001)"
                />
                {/* Show colored border based on error type */}
                {b.error && (
                  <Rect
                    x={rect.x0}
                    y={rect.y0}
                    width={rect.w}
                    height={rect.h}
                    stroke={getErrorStyle(getErrorType(b.error)).strokeColor}
                    strokeWidth={2}
                  />
                )}
                {isSelected && (
                  <Rect
                    x={rect.x0}
                    y={rect.y0}
                    width={rect.w}
                    height={rect.h}
                    stroke="#2563eb"
                    strokeWidth={2}
                    dash={[6, 4]}
                  />
                )}
                {/* simple spinner indicator for pending edits */}
                {isPending && (
                  <Group>
                    <Circle x={centerX} y={centerY} radius={spinnerRadius} stroke="#2563eb" strokeWidth={2} />
                  </Group>
                )}
              </Group>
            )
          })}
        </Layer>
        
      </Stage>
    </div>
  )
}


