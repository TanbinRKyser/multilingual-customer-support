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
  explanation: { word: string; weight: number }[] = [];

  constructor(private http: HttpClient) {}

  sendMessage() {
  const payload = { message: this.message, user_id: 'frontend-user' };

  this.http.post<any>('http://127.0.0.1:8000/chat', payload).subscribe({
      next: (resp) => {
        console.log('Backend response:', resp); // ✅ check in browser console
        this.response = resp.response; // ✅ correct key name
        this.explanation = (resp.explanation || [])
          .map(([word, weight]: [string, number]) => ({ word, weight }));
      },
      error: (err) => {
        console.error('HTTP error:', err);
      }
    });
  }

}
