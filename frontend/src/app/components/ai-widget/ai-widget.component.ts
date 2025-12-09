import { Component, ElementRef, HostListener, ViewChild } from "@angular/core";
import { Router } from "@angular/router";
import { ApiService } from "../../services/api.service";

@Component({
  selector: "app-ai-widget",
  templateUrl: "./ai-widget.component.html",
  styleUrls: ["./ai-widget.component.scss"],
})
export class AiWidgetComponent {
  @ViewChild("historyRef") historyRef?: ElementRef<HTMLDivElement>;
  open = false;
  question = "";
  loading = false;
  history: { role: string; content: string }[] = [];
  panelHeight = 520;
  private dragStartY = 0;
  private dragStartHeight = 0;
  private dragging = false;

  constructor(private api: ApiService, private router: Router) {}

  toggle() {
    this.open = !this.open;
    this.scrollToBottom();
  }

  ask() {
    const prompt = (this.question || "").trim();
    if (!prompt) return;
    this.question = "";
    this.loading = true;
    this.history.push({ role: "user", content: prompt });
    const focus = this.router.url || "global";
    this.api.askAi(prompt, focus, {}, this.history).subscribe({
      next: (resp) => {
        this.history.push({ role: "assistant", content: resp.answer });
        this.loading = false;
        this.scrollToBottom();
      },
      error: () => {
        this.loading = false;
      }
    });
  }

  onEnter(event: KeyboardEvent) {
    if (!event.shiftKey) {
      event.preventDefault();
      this.ask();
    }
  }

  startDrag(event: MouseEvent) {
    this.dragging = true;
    this.dragStartY = event.clientY;
    this.dragStartHeight = this.panelHeight;
    event.preventDefault();
  }

  @HostListener("window:mousemove", ["$event"])
  onMouseMove(event: MouseEvent) {
    if (!this.dragging) return;
    const delta = this.dragStartY - event.clientY;
    this.panelHeight = Math.min(900, Math.max(320, this.dragStartHeight + delta));
  }

  @HostListener("window:mouseup")
  onMouseUp() {
    this.dragging = false;
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      if (this.historyRef?.nativeElement) {
        const el = this.historyRef.nativeElement;
        el.scrollTop = el.scrollHeight;
      }
    }, 50);
  }
}
