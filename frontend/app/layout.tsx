import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'MedMem0 - Longitudinal Patient Memory',
  description: 'Compare retrieval strategies for medical memory',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50">
        <nav className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <a href="/" className="text-xl font-bold text-blue-600">
                MedMem0
              </a>
              <div className="flex gap-6">
                <a href="/" className="text-gray-600 hover:text-blue-600">
                  Search
                </a>
                <a href="/about" className="text-gray-600 hover:text-blue-600">
                  About
                </a>
              </div>
            </div>
          </div>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}