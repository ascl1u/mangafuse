import { create } from 'zustand'

export type Depth = 'cleaned' | 'full'
type EditsMap = Record<number, { en_text?: string }>

type UIStoreState = {
  depth: Depth
  setDepth: (d: Depth) => void

  selectedBubbleId?: number
  setSelectedBubbleId: (id?: number) => void

  edits: EditsMap
  updateEdit: (id: number, patch: { en_text?: string }) => void
}

export const useAppStore = create<UIStoreState>((set, get) => ({
  depth: 'cleaned',
  setDepth: (d) => set({ depth: d }),

  selectedBubbleId: undefined,
  setSelectedBubbleId: (id) => set({ selectedBubbleId: id }),

  edits: {},
  updateEdit: (id, patch) => {
    set({ edits: { ...get().edits, [id]: { ...get().edits[id], ...patch } } })
  },
}))