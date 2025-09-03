export type ProjectStatus = 'PENDING' | 'PROCESSING' | 'TRANSLATING' | 'TYPESETTING' | 'UPDATING' | 'COMPLETED' | 'FAILED'

export type EditorBubble = {
	id: number
	polygon: [number, number][]
	ja_text?: string
	en_text?: string
	font_size?: number
	rect?: { x: number; y: number; w: number; h: number }
	crop_url?: string
	error?: string
}

export type EditorPayload = {
	image_url: string
	width: number
	height: number
	bubbles: EditorBubble[]
	text_layer_url?: string
}

export type PollPayload = {
	project_id: string
	status: ProjectStatus
	task_state?: string
	meta?: { stage?: string; progress?: number }
	error?: string
	editor_data_rev?: number
	artifacts?: { [key: string]: string }
	editor_data?: EditorPayload
	editor_payload_url?: string
}

export type ExportPayload = {
	final_url?: string
	text_layer_url?: string
}

export type ProjectListItem = {
	project_id: string
	title: string
	status: ProjectStatus
	updated_at?: string
}

export type ProjectListResponse = {
	items: ProjectListItem[]
	page: number
	limit: number
	has_next: boolean
}


