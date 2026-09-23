import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

type Source = { source: string; snippet?: string; chunk?: number };
type LimePair = [string, number];
type IGToken = { token: string; weight: number };

interface ChatResponse {
  original_message: string;
  detected_language: string;
  language_name: string;
  intent: string;
  confidence: number;   
  response: string;
  resolution: 'intent' | 'knowledge_base' | 'handoff';
  sources?: Source[] | null;
  explanation?: LimePair[] | IGToken[] | null;
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
  languageName = '';
  intent = '';
  confidence = 0;
  response = '';
  resolution = '';
  sources: Source[] = [];
  explanation: IGToken[] = [];

  loading = false;
  error = '';
  explainMethod: 'lime' | 'ig' | null = 'lime';
  readonly examples = [
    'I forgot my password',
    'Wo ist meine Bestellung?',
    'Je veux annuler ma commande'
  ];

  constructor( private http: HttpClient ) {}

  sendMessage() {
    if ( !this.message.trim() ) return;

    this.loading = true;
    this.error = '';
    this.response = '';

    const payload: { message: string; explain_method?: 'lime' | 'ig' } = {
      message: this.message,
      ...(this.explainMethod ? { explain_method: this.explainMethod } : {})
    };

    this.http.post<ChatResponse>('http://127.0.0.1:8000/chat', payload).subscribe({
      next: (resp) => {
        this.originalMessage   = resp.original_message ?? '';
        this.detectedLanguage  = resp.detected_language ?? '';
        this.languageName      = resp.language_name ?? resp.detected_language ?? '';
        this.intent            = resp.intent ?? '';
        this.confidence        = resp.confidence ?? 0;
        this.response          = resp.response ?? '';
        this.resolution        = resp.resolution ?? '';
        this.sources           = resp.sources ?? [];

        const exp = resp.explanation ?? [];
        if (Array.isArray(exp) && exp.length > 0) {
          if (Array.isArray(exp[0])) {
            // LIME format: [["word", weight], ...]
            this.explanation = (exp as LimePair[]).map(([token, weight]) => ({ token, weight }));
          } else {
            // IG format: [{token, weight}, ...]
            this.explanation = (exp as IGToken[]);
          }
        } else {
          this.explanation = [];
        }

        this.loading = false;
      },
      error: ( err ) => {
        this.error = 'Request failed. Check the backend.';
        console.error( err );
        this.loading = false;
      }
    });
  }

  useExample(example: string) {
    this.message = example;
    this.sendMessage();
  }

  onKeydown(event: KeyboardEvent) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') this.sendMessage();
  }

  confidencePct(): number {
    return Math.round( ( this.confidence || 0 ) * 100 );
  }

  weightColor(w: number): string {
    const a = Math.min( Math.abs( w ), 1);
    return w >= 0 ? `rgba(30,111,88,${.12 + a * .35})` : `rgba(200,70,50,${.12 + a * .35})`;
  }

}
