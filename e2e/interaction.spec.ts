import { test, expect, type Page, type Locator } from '@playwright/test';

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(`console.error: ${msg.text()}`);
  });
  page.on('requestfailed', (req) => {
    errors.push(`requestfailed: ${req.url()} ${req.failure()?.errorText ?? ''}`);
  });
  return errors;
}

async function selectOption(page: Page, testId: string, optionText: string | RegExp) {
  await page.getByTestId(testId).click();
  // Force-click the option to avoid flaky overlays on mobile viewports.
  await page.getByRole('option', { name: optionText }).click({ force: true });
}

async function setSliderValue(page: Page, testId: string, value: number, step = 0.1) {
  const thumb = page.locator(`[data-testid="${testId}"] [data-slot="slider-thumb"]`);
  await thumb.waitFor({ state: 'visible' });
  await thumb.focus();

  // Move to the maximum value first, then step back to the target.
  await page.keyboard.press('End');
  const max = Number(await thumb.getAttribute('aria-valuemax'));
  const min = Number(await thumb.getAttribute('aria-valuemin'));
  const target = Math.max(min, Math.min(max, value));
  const leftSteps = Math.round((max - target) / step);
  for (let i = 0; i < leftSteps; i++) {
    await page.keyboard.press('ArrowLeft');
  }
}

async function assertChartHasData(chartLocator: Locator) {
  const svg = chartLocator.locator('svg').first();
  await expect(svg).toBeVisible();
  const paths = await chartLocator.locator('path').count();
  expect(paths).toBeGreaterThan(0);
}

test.describe('chapter 08 FA high-risk interactions', () => {
  test('semi-gradient TD charts update when feature and weight mode change', async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto('/#/ch08/fa');
    await page.waitForLoadState('domcontentloaded');

    await page.getByRole('tab', { name: /半梯度 TD/ }).click();
    await expect(page.getByRole('tab', { name: /半梯度 TD/ })).toHaveAttribute('aria-selected', 'true');

    for (const feature of ['坐标归一化', '坐标多项式', '距离目标']) {
      await selectOption(page, 'fa-feature-select', feature);
      await expect(page.getByTestId('fa-value-error-chart')).toBeVisible();
      await assertChartHasData(page.getByTestId('fa-value-error-chart'));
      await assertChartHasData(page.getByTestId('fa-fixed-point-chart'));
      await assertChartHasData(page.getByTestId('fa-objective-chart'));
    }

    for (const mode of ['empirical visitation', 'stationary']) {
      await page.getByRole('button', { name: mode }).click();
      await assertChartHasData(page.getByTestId('fa-objective-chart'));
    }

    expect(errors).toEqual([]);
  });

  test('LSTD panel shows success status and numeric diagnostics', async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto('/#/ch08/fa');
    await page.waitForLoadState('domcontentloaded');

    await page.getByRole('tab', { name: '理论' }).click();
    await page.getByRole('tab', { name: 'LSTD' }).click();
    await expect(page.getByRole('tab', { name: 'LSTD' })).toHaveAttribute('aria-selected', 'true');

    await expect(page.getByTestId('lstd-status')).toHaveText('成功');
    await expect(page.getByTestId('lstd-ridge')).not.toHaveText('—');
    await expect(page.getByTestId('lstd-min-pivot')).not.toBeEmpty();
    await expect(page.getByTestId('lstd-cond')).not.toBeEmpty();

    for (const feature of ['coordinate', 'polynomial', 'distance']) {
      await selectOption(page, 'lstd-feature-select', feature);
      await expect(page.getByTestId('lstd-status')).toHaveText('成功');
    }

    expect(errors).toEqual([]);
  });

  test('DQN update-level and episode-level charts split metrics', async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto('/#/ch08/fa');
    await page.waitForLoadState('domcontentloaded');

    await page.getByRole('tab', { name: /Deep Q-learning/ }).click();
    await expect(page.getByRole('tab', { name: /Deep Q-learning/ })).toHaveAttribute('aria-selected', 'true');

    await assertChartHasData(page.getByTestId('dqn-update-chart'));
    await assertChartHasData(page.getByTestId('dqn-episode-chart'));

    await selectOption(page, 'dqn-task-select', /continuing/);
    await assertChartHasData(page.getByTestId('dqn-update-chart'));
    await assertChartHasData(page.getByTestId('dqn-episode-chart'));

    await selectOption(page, 'dqn-task-select', /episodic/);
    await assertChartHasData(page.getByTestId('dqn-update-chart'));
    await assertChartHasData(page.getByTestId('dqn-episode-chart'));

    expect(errors).toEqual([]);
  });
});

test.describe('chapter 09 PG high-risk interactions', () => {
  test('softmax shift invariance preserves probabilities', async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto('/#/ch09/pg');
    await page.waitForLoadState('domcontentloaded');

    await page.getByRole('tab', { name: /策略函数/ }).click();
    await expect(page.getByRole('tab', { name: /策略函数/ })).toHaveAttribute('aria-selected', 'true');

    const initialProbs = await page.getByTestId('pg-shift-probs').textContent();
    expect(initialProbs).toBeTruthy();

    await setSliderValue(page, 'pg-shift-slider', 3);
    const shiftedProbs = await page.getByTestId('pg-shift-probs').textContent();
    expect(shiftedProbs).toBe(initialProbs);

    expect(errors).toEqual([]);
  });

  test('changing Δθ updates affected-state logits', async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto('/#/ch09/pg');
    await page.waitForLoadState('domcontentloaded');

    await page.getByRole('tab', { name: /策略函数/ }).click();

    const before = await page.getByTestId('pg-delta-logits').textContent();
    await setSliderValue(page, 'pg-delta-theta-slider', 1.5);
    const after = await page.getByTestId('pg-delta-logits').textContent();
    expect(after).not.toBe(before);

    expect(errors).toEqual([]);
  });
});
