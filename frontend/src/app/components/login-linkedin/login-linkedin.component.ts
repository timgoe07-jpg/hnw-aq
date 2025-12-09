import { Component, OnInit } from "@angular/core";
import { ActivatedRoute, Router } from "@angular/router";
import { MatSnackBar } from "@angular/material/snack-bar";
import { ApiService } from "../../services/api.service";

@Component({
  selector: "app-login-linkedin",
  templateUrl: "./login-linkedin.component.html",
  styleUrls: ["./login-linkedin.component.scss"],
})
export class LoginLinkedInComponent implements OnInit {
  status = "Processing...";

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private api: ApiService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    const code = this.route.snapshot.queryParamMap.get("code");
    const state = this.route.snapshot.queryParamMap.get("state");
    if (!code || !state) {
      this.status = "Missing OAuth parameters.";
      return;
    }
    this.api.handleLinkedInCallback(code, state).subscribe({
      next: () => {
        this.status = "LinkedIn connected.";
        this.snackBar.open("LinkedIn connected", "Close", { duration: 2500 });
        setTimeout(() => this.router.navigate(["/match"]), 500);
      },
      error: (e) => {
        console.error(e);
        this.status = "Failed to connect LinkedIn.";
        this.snackBar.open("LinkedIn auth failed", "Close", { duration: 2500 });
      },
    });
  }
}
