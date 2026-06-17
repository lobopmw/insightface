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
  const setCurrentPage = useNavigationStore((state) => state.setCurrentPage);
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const user = useAuthStore((state) => state.user);
  const restoreSession = useAuthStore((state) => state.restoreSession);

  useEffect(() => {
    void restoreSession();
  }, [restoreSession]);

  useEffect(() => {
    if (user?.role !== "admin" && currentPage === "students") {
      setCurrentPage("dashboard");
    }
  }, [currentPage, setCurrentPage, user?.role]);

  if (!isAuthenticated) {
    return <Login />;
  }

  return <AppLayout>{pages[currentPage]}</AppLayout>;
}
