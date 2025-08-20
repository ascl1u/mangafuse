/**
 * Abstract interface for an file uploader.
 */
interface Uploader {
  upload(file: File, uploadUrl: string, headers: Record<string, string>): Promise<void>;
}

/**
 * Uploader for production, which uploads directly to a presigned S3 URL.
 */
class ProductionUploader implements Uploader {
  async upload(file: File, uploadUrl: string): Promise<void> {
    const resp = await fetch(uploadUrl, {
      method: 'PUT',
      body: file,
      headers: { 'Content-Type': file.type },
    });
    if (!resp.ok) {
      throw new Error(`Upload failed (${resp.status})`);
    }
  }
}

const API_BASE = import.meta.env.VITE_API_BASE;

/**
 * Uploader for local development, which posts to a local backend endpoint.
 */
class DevelopmentUploader implements Uploader {
  async upload(file: File, uploadUrl: string, headers: Record<string, string>): Promise<void> {
    const form = new FormData();
    form.append('file', file);
    const resp = await fetch(`${API_BASE}${uploadUrl}`, {
      method: 'POST',
      body: form,
      headers,
    });
    if (!resp.ok) {
      throw new Error('Upload failed');
    }
  }
}

/**
 * Factory function to get the appropriate uploader based on the environment.
 */
export function getUploader(): Uploader {
  if (import.meta.env.MODE === 'production') {
    return new ProductionUploader();
  }
  return new DevelopmentUploader();
}
