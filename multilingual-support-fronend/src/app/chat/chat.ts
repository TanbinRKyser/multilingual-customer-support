import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

type Source = { source: string; snippet?: string; chunk?: number };

interface ChatResponse {
  original_message: string;
  detected_language: string;
  intent: string;
  confidence: number;   // 0..1
  response: string;
  sources?: Source[] | null;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.html',
  styleUrls: ['./chat.css']
})
export class Chat {
  message = '';

  // from backend
  originalMessage = '';
  detectedLanguage = '';
  intent = '';
  confidence = 0;
  response = '';
  sources: Source[] = [];

  loading = false;
  error = '';

  constructor(private http: HttpClient) {}

  sendMessage() {
    if (!this.message.trim()) return;

    this.loading = true;
    this.error = '';
    const payload = { message: this.message };

    this.http.post<ChatResponse>('http://127.0.0.1:8000/chat', payload).subscribe({
      next: (resp) => {
        this.originalMessage   = resp.original_message ?? '';
        this.detectedLanguage  = resp.detected_language ?? '';
        this.intent            = resp.intent ?? '';
        this.confidence        = resp.confidence ?? 0;
        this.response          = resp.response ?? '';
        this.sources           = resp.sources ?? [];
        this.loading           = false;
      },
      error: (err) => {
        this.error = 'Request failed. Check the backend.';
        console.error(err);
        this.loading = false;
      }
    });
  }

  confidencePct(): number {
    return Math.round((this.confidence || 0) * 100);
  }
}
