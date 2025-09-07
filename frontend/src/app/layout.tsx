import "./globals.css";

export const metadata = {
  title: "Retail Analytics – Demo",
  description: "Upload a video to get people counts, peak snapshot, and heatmap analysis.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-neutral-950 text-neutral-100 antialiased">
        {children}
      </body>
    </html>
  );
}
