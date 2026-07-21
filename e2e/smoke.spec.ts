import { test, expect, type Page } from '@playwright/test';

const routes = [
  '/',
  '/#/ch01/overview',
  '/#/ch01/mdp',
  '/#/ch01/policy',
  '/#/ch01/reward',
  '/#/ch01/returns',
  '/#/ch02/overview',
  '/#/ch02/bellman',
  '/#/ch02/state-values',
  '/#/ch02/action-values',
  '/#/ch03/overview',
  '/#/ch03/boe',
  '/#/ch04/overview',
  '/#/ch04/vi-pi',
  '/#/ch04/convergence',
  '/#/ch05/overview',
  '/#/ch05/mc',
  '/#/ch05/off-policy',
  '/#/ch06/overview',
  '/#/ch06/sa',
  '/#/ch07/overview',
  '/#/ch07/td',
  '/#/ch07/td-ext',
  '/#/ch08/overview',
  '/#/ch08/fa',
  '/#/ch09/overview',
  '/#/ch09/pg',
  '/#/ch10/overview',
  '/#/ch10/ac',
];

function isExternalTransientError(url: string, errorText: string): boolean {
  // Network hiccups for external fonts/styles are not application bugs.
  if (url.includes('fonts.googleapis.com') || url.includes('fonts.gstatic.com')) return true;
  if (errorText.includes('ERR_CONNECTION_CLOSED') || errorText.includes('ERR_CONNECTION_RESET')) return true;
  return false;
}

async function collectErrors(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      const text = msg.text();
      if (text.includes('fonts.googleapis.com') || text.includes('fonts.gstatic.com')) return;
      // Generic Chromium resource-load failures duplicate the requestfailed
      // events we already track (and filter) above.
      if (/Failed to load resource: net::ERR_/.test(text)) return;
      errors.push(`console.error: ${text}`);
    }
  });
  page.on('requestfailed', (req) => {
    const url = req.url();
    const errorText = req.failure()?.errorText ?? '';
    if (isExternalTransientError(url, errorText)) return;
    errors.push(`requestfailed: ${url} ${errorText}`);
  });
  page.on('pageerror', (err) => {
    // also capture unhandled promise rejections surfaced as page errors
    if (err.message.includes('unhandled')) {
      errors.push(`unhandled: ${err.message}`);
    }
  });
  return errors;
}

async function safeClickTab(tab: ReturnType<Page['locator']>) {
  await tab.scrollIntoViewIfNeeded();
  try {
    await tab.click({ force: true });
  } catch {
    // Fall back to a programmatic click when the tab is off-screen in a
    // scrollable tablist on small viewports.
    await tab.evaluate((el) => el.click());
  }
}

for (const route of routes) {
  test(`route ${route} renders without runtime errors`, async ({ page }) => {
    const errors = await collectErrors(page);
    await page.goto(route);
    await page.waitForLoadState('domcontentloaded');

    // Page should not be blank.
    const bodyText = await page.locator('body').innerText();
    expect(bodyText.length).toBeGreaterThan(0);

    // Main interactive area should exist.
    await expect(page.locator('main, [class*="max-w"], .prose').first()).toBeVisible();

    // Click every tab trigger and ensure no errors are emitted. Re-scan after
    // each click so newly-visible nested tabs are also exercised.
    const clicked = new Set<string>();
    let changed = true;
    while (changed) {
      changed = false;
      const tabs = page.locator('[role="tab"]');
      const count = await tabs.count();
      for (let i = 0; i < count; i++) {
        const tab = tabs.nth(i);
        const id = (await tab.getAttribute('id')) ?? (await tab.textContent()) ?? `tab-${i}`;
        const key = `${i}-${id}`;
        if (clicked.has(key)) continue;

        await safeClickTab(tab);
        await expect(tab).toHaveAttribute('aria-selected', 'true');
        const controls = await tab.getAttribute('aria-controls');
        if (controls) {
          await expect(page.locator(`#${controls}`)).toBeVisible();
        }
        clicked.add(key);
        changed = true;
      }
    }

    expect(errors).toEqual([]);
  });
}
