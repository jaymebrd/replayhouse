-- Reward by day and task family (Grafana: time series, one line per family)
SELECT toDate(inserted_at) AS day, task_family, round(avg(reward), 3) AS mean_reward
FROM trajectories GROUP BY day, task_family ORDER BY day, task_family;

-- Worst 10 trajectories to eyeball (Grafana: table panel)
SELECT task_family, answer, reward, total_tokens
FROM trajectories ORDER BY reward ASC, total_tokens DESC LIMIT 10;

-- Token spend per family (Grafana: bar gauge)
SELECT task_family, sum(total_tokens) AS tokens, count() AS episodes
FROM trajectories GROUP BY task_family ORDER BY tokens DESC;

-- Success rate overall and last day (Grafana: stat panels)
SELECT round(avg(reward), 3) AS overall,
       round(avgIf(reward, inserted_at >= now() - INTERVAL 1 DAY), 3) AS last_day
FROM trajectories;

-- Current priority distribution (Grafana: histogram, uses sidecar)
SELECT p, count() AS n FROM
(
    SELECT id, round(argMax(priority, version), 1) AS p
    FROM trajectories__priorities GROUP BY id
)
GROUP BY p ORDER BY p;

-- Store size: rows and bytes on disk (Grafana: stat panels)
SELECT sum(rows) AS rows, formatReadableSize(sum(bytes_on_disk)) AS size
FROM system.parts WHERE active AND database = currentDatabase()
  AND `table` = 'trajectories';
