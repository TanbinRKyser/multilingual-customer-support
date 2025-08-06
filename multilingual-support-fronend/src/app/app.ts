import { Component, signal } from '@angular/core';
import { Chat } from './chat/chat'; // ✅ correct path

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [Chat],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App {
  protected readonly title = signal('frontend');
}
