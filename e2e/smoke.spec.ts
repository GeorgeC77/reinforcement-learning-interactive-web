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

async function collectErrors(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      errors.push(`console.error: ${msg.text()}`);
    }
  });
  return errors;
}

for (const route of routes) {
  test(`route ${route} renders without runtime errors`, async ({ page }) => {
    const errors = await collectErrors(page);
    await page.goto(route);
    await page.waitForLoadState('networkidle');

    // Page should not be blank.
    const bodyText = await page.locator('body').innerText();
    expect(bodyText.length).toBeGreaterThan(0);

    // Main interactive area should exist.
    await expect(page.locator('main, [class*="max-w"], .prose').first()).toBeVisible();

    // Click every tab trigger and ensure no errors are emitted.
    // On narrow mobile viewports some tabs may overflow their container;
    // fall back to a programmatic click so we still exercise the content.
    const tabs = page.locator('[role="tab"]');
    const count = await tabs.count();
    for (let i = 0; i < count; i++) {
      const tab = tabs.nth(i);
      await tab.scrollIntoViewIfNeeded();
      try {
        await tab.click({ force: true, timeout: 3000 });
      } catch {
        await tab.evaluate((el: HTMLElement) => el.click());
      }
      await page.waitForTimeout(150);
    }

    expect(errors).toEqual([]);
  });
}
