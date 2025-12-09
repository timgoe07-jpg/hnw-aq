import { Component, OnInit } from "@angular/core";
import { MatSnackBar } from "@angular/material/snack-bar";
import { ApiService } from "../../services/api.service";
import { Persona } from "../../models/persona.model";

@Component({
  selector: "app-persona-list",
  templateUrl: "./persona-list.component.html",
  styleUrls: ["./persona-list.component.scss"],
})
export class PersonaListComponent implements OnInit {
  personas: Persona[] = [];
  loading = false;

  constructor(private api: ApiService, private snackBar: MatSnackBar) {}

  ngOnInit(): void {
    this.fetch();
  }

  fetch(): void {
    this.loading = true;
    this.api.getPersonas().subscribe({
      next: (data) => {
        this.personas = data;
        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
        this.snackBar.open("Failed to load personas", "Close", { duration: 2500 });
      },
    });
  }
}
