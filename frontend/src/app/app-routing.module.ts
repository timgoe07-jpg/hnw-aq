import { NgModule } from "@angular/core";
import { RouterModule, Routes } from "@angular/router";
import { DashboardComponent } from "./components/dashboard/dashboard.component";
import { PredictionsComponent } from "./components/predictions/predictions.component";
import { ReportComponent } from "./components/report/report.component";
import { PersonaListComponent } from "./components/persona-list/persona-list.component";
import { CaseStudiesComponent } from "./components/case-studies/case-studies.component";
import { ProfileUploadComponent } from "./components/profile-upload/profile-upload.component";
import { LoginLinkedInComponent } from "./components/login-linkedin/login-linkedin.component";
import { CalculatorComponent } from "./components/calculator/calculator.component";

const routes: Routes = [
  { path: "dashboard", component: DashboardComponent },
  { path: "predictions", component: PredictionsComponent },
  { path: "report", component: ReportComponent },
  { path: "personas", component: PersonaListComponent },
  { path: "case-studies", component: CaseStudiesComponent },
  { path: "match", component: ProfileUploadComponent },
  { path: "auth/linkedin/callback", component: LoginLinkedInComponent },
  { path: "calculator", component: CalculatorComponent },
  { path: "", pathMatch: "full", redirectTo: "dashboard" },
  { path: "**", redirectTo: "dashboard" }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}
