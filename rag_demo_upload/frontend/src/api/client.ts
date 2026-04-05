export interface SimpleContextItem {
  id: string;
}

export interface SimpleRAGResponse {
  ok: boolean;
  answer: string;
  explanation: string;
  related_count: number;
  related_items: SimpleContextItem[];
}

export async function askQuestion(question: string): Promise<SimpleRAGResponse> {
  const res = await fetch("/api/rag/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });

  if (!res.ok) {
    throw new Error(`请求失败：HTTP ${res.status}`);
  }

  return res.json();
}

