import { Component } from '@angular/core';
import { Chat } from './chat/chat';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [Chat],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App {}
