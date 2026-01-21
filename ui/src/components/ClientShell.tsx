"use client";

import { usePathname } from "next/navigation";
import Sidebar from "@/components/Sidebar";

 function ClientShell({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const isLandingPage = pathname === "/";

  return (
    <div className="flex min-h-screen bg-black">
      {!isLandingPage && <Sidebar />}
      <main className="flex-1 overflow-x-hidden">
        {children}
      </main>
    </div>
  );
}

export default ClientShell
