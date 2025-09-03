import { createFileRoute, Link } from '@tanstack/react-router'
import type { ProjectListResponse, ProjectListItem } from '../../types'
import { listProjects } from '../../api'

export const Route = createFileRoute('/projects/')({
  loader: async ({ context, location }) => {
    const { getToken } = (context as { getToken: () => Promise<string | null> })
    const search = new URLSearchParams(location.searchStr)
    const page = Number(search.get('page') || '1')
    const limit = Number(search.get('limit') || '20')
    const token = await getToken()
    if (!token) {
      return { items: [], page, limit, has_next: false, requiresAuth: true } as ProjectListResponse & { requiresAuth: boolean }
    }
    const data = await listProjects(getToken, page, limit)
    return data
  },
  component: ProjectsPage,
})

function ProjectsPage() {
  const data = Route.useLoaderData() as ProjectListResponse & { requiresAuth?: boolean }
  if (data?.requiresAuth) {
    return <div className="text-sm text-gray-600">Please sign in to view your projects.</div>
  }
  const items = data?.items || []
  const page = data?.page || 1
  const limit = data?.limit || 20
  const hasNext = !!data?.has_next
  const prevPage = Math.max(1, page - 1)
  const nextPage = page + 1
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">My Projects</h1>
      </div>
      {items.length === 0 ? (
        <div className="text-sm text-gray-600">No projects found.</div>
      ) : (
        <div className="overflow-x-auto bg-white border rounded">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 text-left">
              <tr>
                <th className="px-3 py-2 font-medium text-gray-600">Title</th>
                <th className="px-3 py-2 font-medium text-gray-600">Status</th>
                <th className="px-3 py-2 font-medium text-gray-600">Updated</th>
              </tr>
            </thead>
            <tbody>
              {items.map((p: ProjectListItem) => (
                <tr key={p.project_id} className="border-t">
                  <td className="px-3 py-2">
                    <Link to="/projects/$projectId" params={{ projectId: p.project_id }} className="text-blue-600 hover:underline">
                      {p.title || p.project_id}
                    </Link>
                  </td>
                  <td className="px-3 py-2">{p.status}</td>
                  <td className="px-3 py-2">{p.updated_at ? new Date(p.updated_at).toLocaleString() : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="flex items-center justify-between">
        <Link
          to="/projects"
          search={{ page: prevPage, limit }}
          className="px-3 py-2 rounded border bg-white text-gray-700 disabled:opacity-50"
        >
          Prev
        </Link>
        <div className="text-sm text-gray-600">Page {page}</div>
        {hasNext ? (
          <Link
            to="/projects"
            search={{ page: nextPage, limit }}
            className="px-3 py-2 rounded border bg-white text-gray-700"
          >
            Next
          </Link>
        ) : (
          <button className="px-3 py-2 rounded border bg-white text-gray-400 cursor-not-allowed" disabled>
            Next
          </button>
        )}
      </div>
    </div>
  )
}


