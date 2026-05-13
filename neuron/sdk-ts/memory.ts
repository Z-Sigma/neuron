import axios from 'axios';

export interface Node {
  id: string;
  label: string;
  confidence: number;
  evidence_count: number;
}

export interface RetrievalContext {
  direct_beliefs: Node[];
  unresolved_tensions: any[];
  related_context: Node[];
}

export class Memory {
  private baseUrl: string;
  private userId: string;

  constructor(userId: string, baseUrl: string = 'http://localhost:8000') {
    this.userId = userId;
    this.baseUrl = baseUrl;
  }

  async process(text: string): Promise<Node | null> {
    const response = await axios.post(`${this.baseUrl}/v1/memory/process`, {
      text,
      user_id: this.userId
    });
    return response.data.node;
  }

  async retrieve(query: string): Promise<RetrievalContext> {
    const response = await axios.post(`${this.baseUrl}/v1/memory/retrieve`, {
      query,
      user_id: this.userId
    });
    return response.data.context;
  }

  async consolidate(): Promise<void> {
    await axios.post(`${this.baseUrl}/v1/memory/consolidate/${this.userId}`);
  }
}
