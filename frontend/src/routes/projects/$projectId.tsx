import { createFileRoute } from '@tanstack/react-router'
import { fetchProjectById } from '../../api'
type RouterContext = { getToken: () => Promise<string | null> }

function ProjectPage() {
  const { projectId } = Route.useParams()
  const data = Route.useLoaderData()
  const status = data?.status
  const stage = data?.meta?.stage
  return (
    <div className="max-w-2xl">
      <h1 className="text-xl font-semibold mb-3">Project {projectId}</h1>
      <div className="text-sm">Status: {status}{stage ? ` — ${stage}` : ''}</div>
    </div>
  )
}

function ProjectError({ error }: { error: unknown }) {
  const { projectId } = Route.useParams()
  const message = error instanceof Error ? error.message : 'Unknown error'
  return (
    <div className="max-w-2xl">
      <h1 className="text-xl font-semibold mb-3">Project {projectId}</h1>
      <div className="p-3 rounded border border-red-200 bg-red-50 text-sm text-red-700">{message}</div>
    </div>
  )
}

export const Route = createFileRoute('/projects/$projectId')({
  loader: async ({ params, context }) => {
    console.log('[loader] fetching project', params.projectId)
    const token = await (context as RouterContext).getToken()
    const data = await fetchProjectById(params.projectId, token || undefined)
    return data
  },
  errorComponent: ProjectError,
  component: ProjectPage,
})


