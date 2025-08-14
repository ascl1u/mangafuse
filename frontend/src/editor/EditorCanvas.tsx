import { useEffect, useMemo, useRef, useState } from 'react'
import { Stage, Layer, Image as KonvaImage, Line, Text as KonvaText, Group } from 'react-konva'
import type Konva from 'konva'
import { loadWebFontOnce } from './useWebFont'
import { FONT_FAMILY, FONT_URL, FONT_SIZE_MIN } from '../constants'
import type { EditorPayload, EditorBubble } from '../store'

type Props = {
  editor: EditorPayload
  selectedId?: number
  onSelect: (id: number) => void
  edits: Record<number, { en_text?: string; font_size?: number }>
  disabled?: boolean
}

function useHtmlImage(src: string): HTMLImageElement | undefined {
  const [img, setImg] = useState<HTMLImageElement>()
  useEffect(() => {
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

function polygonToLinePoints(poly: [number, number][]): number[] {
  const pts: number[] = []
  for (const [x, y] of poly) {
    pts.push(x, y)
  }
  return pts
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

export function EditorCanvas({ editor, selectedId, onSelect, edits, disabled }: Props) {
  const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
  const imageUrl = `${API_BASE}${editor.image_url}`
  const img = useHtmlImage(imageUrl)
  // Load Anime Ace font from assets for client-side overlay
  const fontUrl = `${API_BASE}${FONT_URL}`
  const [fontReady, setFontReady] = useState<boolean>(false)
  useEffect(() => {
    setFontReady(false)
    loadWebFontOnce(FONT_FAMILY, fontUrl).finally(() => setFontReady(true))
  }, [fontUrl])

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

  const aspect = editor.height / editor.width
  const stageWidth = Math.max(100, containerWidth)
  const stageHeight = Math.max(100, Math.round(stageWidth * aspect))

  // Scale from image coords to stage coords
  const scale = stageWidth / editor.width

  const bubbles = editor.bubbles as EditorBubble[]

  const polygons = useMemo(() => {
    return bubbles.map((b) => b.polygon.map(([x, y]) => [x * scale, y * scale]) as [number, number][])
  }, [bubbles, scale])

  const selectedIdx = bubbles.findIndex((b) => b.id === selectedId)

  return (
    <div ref={containerRef} className="w-full">
      <Stage width={stageWidth} height={stageHeight} listening={!disabled}>
        <Layer listening={false}>
          {img && <KonvaImage image={img} width={stageWidth} height={stageHeight} />}
        </Layer>
        <Layer>
          {polygons.map((poly, idx) => {
            const b = bubbles[idx]
            const isSelected = idx === selectedIdx
            const baseColor = isSelected ? '#2563eb' /* blue-600 */ : 'rgba(0,0,0,0.35)'
            const fill = isSelected ? 'rgba(37,99,235,0.15)' : 'rgba(0,0,0,0.08)'
            const points = polygonToLinePoints(poly)
            const onEnter = (e: Konva.KonvaEventObject<MouseEvent>) => {
              const node = e.target
              if (!isSelected) node.opacity(0.9)
            }
            const onLeave = (e: Konva.KonvaEventObject<MouseEvent>) => {
              const node = e.target
              if (!isSelected) node.opacity(1)
            }
            // live text
            const edit = edits[b.id] || {}
            const text = (edit.en_text ?? b.en_text ?? '').trim()
            const rect = b.rect
              ? { x0: b.rect.x * scale, y0: b.rect.y * scale, w: Math.max(1, b.rect.w * scale), h: Math.max(1, b.rect.h * scale) }
              : polygonBBox(poly)
            return (
              <Group key={b.id} onClick={() => !disabled && onSelect(b.id)}>
                <Line
                  points={points}
                  closed
                  stroke={baseColor}
                  strokeWidth={isSelected ? 2 : 1}
                  fill={fill}
                  onMouseEnter={onEnter}
                  onMouseLeave={onLeave}
                />
                {text && fontReady && (
                  <Group
                    clipFunc={(ctx) => {
                      ctx.beginPath()
                      for (let i = 0; i < points.length; i += 2) {
                        const x = points[i]
                        const y = points[i + 1]
                        if (i === 0) ctx.moveTo(x, y)
                        else ctx.lineTo(x, y)
                      }
                      ctx.closePath()
                    }}
                  >
                    <KonvaText
                      text={text}
                      x={rect.x0}
                      y={rect.y0}
                      width={rect.w}
                      height={rect.h}
                      fontSize={Math.max(FONT_SIZE_MIN, edit.font_size ?? b.font_size ?? 18)}
                      fontFamily={`${FONT_FAMILY}, Arial, sans-serif`}
                      align="center"
                      verticalAlign="middle"
                      fill="#000"
                    />
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


