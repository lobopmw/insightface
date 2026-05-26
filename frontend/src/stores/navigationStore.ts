import { create } from "zustand";

import type { AppPage } from "@/components/layout/AppLayout";

type NavigationState = {
  currentPage: AppPage;
  setCurrentPage: (page: AppPage) => void;
};

export const useNavigationStore = create<NavigationState>((set) => ({
  currentPage: "dashboard",
  setCurrentPage: (page) => set({ currentPage: page }),
}));
