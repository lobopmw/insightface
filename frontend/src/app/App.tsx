import { useEffect } from "react";

import { Dashboard } from "@/pages/Dashboard";
import { Login } from "@/pages/Login";
import { Monitoring } from "@/pages/Monitoring";
import { Reports } from "@/pages/Reports";
import { Settings } from "@/pages/Settings";
import { Students } from "@/pages/Students";
import { AppLayout, type AppPage } from "@/components/layout/AppLayout";
import { useAuthStore } from "@/stores/authStore";
import { useNavigationStore } from "@/stores/navigationStore";

const pages: Record<AppPage, JSX.Element> = {
  dashboard: <Dashboard />,
  monitoring: <Monitoring />,
  students: <Students />,
  reports: <Reports />,
  settings: <Settings />,
};

export function App() {
  const currentPage = useNavigationStore((state) => state.currentPage);
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const restoreSession = useAuthStore((state) => state.restoreSession);

  useEffect(() => {
    void restoreSession();
  }, [restoreSession]);

  if (!isAuthenticated) {
    return <Login />;
  }

  return <AppLayout>{pages[currentPage]}</AppLayout>;
}
