import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.html',
  styleUrls: ['./chat.css']
})
export class Chat {
  message = '';
  response = '';
  originalMessage = '';
  detectedLanguage = '';
  intent = '';
  confidence = 0;
  explanation = '';

  constructor(private http: HttpClient) {}

  sendMessage() {
    const payload = { message: this.message };

    this.http.post<any>('http://127.0.0.1:8000/chat', payload).subscribe({
      next: (resp) => {
        console.log('Backend response:', resp);

        this.originalMessage = resp.original_message || '';
        this.detectedLanguage = resp.detected_language || '';
        this.intent = resp.intent || '';
        this.confidence = resp.confidence || 0;
        this.response = resp.response || '';
        this.explanation = resp.explanation || '';
      },
      error: (err) => {
        console.error('HTTP error:', err);
      }
    });
  }
}
