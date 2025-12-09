import { NgModule } from "@angular/core";
import { BrowserModule } from "@angular/platform-browser";
import { BrowserAnimationsModule } from "@angular/platform-browser/animations";
import { FormsModule, ReactiveFormsModule } from "@angular/forms";
import { HttpClientModule } from "@angular/common/http";

import { AppRoutingModule } from "./app-routing.module";
import { AppComponent } from "./app.component";
import { DashboardComponent } from "./components/dashboard/dashboard.component";
import { PredictionsComponent } from "./components/predictions/predictions.component";
import { ReportComponent } from "./components/report/report.component";
import { PersonaListComponent } from "./components/persona-list/persona-list.component";
import { CaseStudiesComponent } from "./components/case-studies/case-studies.component";
import { ProfileUploadComponent } from "./components/profile-upload/profile-upload.component";
import { MatchResultsComponent } from "./components/match-results/match-results.component";
import { LoginLinkedInComponent } from "./components/login-linkedin/login-linkedin.component";
import { CalculatorComponent } from "./components/calculator/calculator.component";
import { AiWidgetComponent } from "./components/ai-widget/ai-widget.component";

import { MatToolbarModule } from "@angular/material/toolbar";
import { MatButtonModule } from "@angular/material/button";
import { MatIconModule } from "@angular/material/icon";
import { MatCardModule } from "@angular/material/card";
import { MatSnackBarModule } from "@angular/material/snack-bar";
import { MatFormFieldModule } from "@angular/material/form-field";
import { MatInputModule } from "@angular/material/input";
import { MatSelectModule } from "@angular/material/select";
import { MatProgressSpinnerModule } from "@angular/material/progress-spinner";
import { MatDividerModule } from "@angular/material/divider";
import { MatTableModule } from "@angular/material/table";
import { MatSortModule } from "@angular/material/sort";
import { MatPaginatorModule } from "@angular/material/paginator";
import { NgChartsModule } from "ng2-charts";
import { MatButtonToggleModule } from "@angular/material/button-toggle";
import { MatSliderModule } from "@angular/material/slider";
import { MatTabsModule } from "@angular/material/tabs";

@NgModule({
  declarations: [
    AppComponent,
    DashboardComponent,
    PredictionsComponent,
    ReportComponent,
    PersonaListComponent,
    CaseStudiesComponent,
    ProfileUploadComponent,
    MatchResultsComponent,
    LoginLinkedInComponent,
    CalculatorComponent,
    AiWidgetComponent,
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    AppRoutingModule,
    FormsModule,
    ReactiveFormsModule,
    HttpClientModule,
    MatToolbarModule,
    MatButtonModule,
    MatIconModule,
    MatCardModule,
    MatTableModule,
    MatSnackBarModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatProgressSpinnerModule,
    MatDividerModule,
    MatTableModule,
    MatSortModule,
    MatPaginatorModule,
    NgChartsModule,
    MatButtonToggleModule,
    MatSliderModule,
    MatTabsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}
