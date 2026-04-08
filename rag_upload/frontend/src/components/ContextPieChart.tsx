import React from "react";
import ReactECharts from "echarts-for-react";
import type { SimpleContextItem } from "../api/client";

interface Props {
  items: SimpleContextItem[];
}

// 用饼图展示“有无参考资料”这个概念，小白也能看懂
export const ContextPieChart: React.FC<Props> = ({ items }) => {
  const hasContext = items.length > 0;

  const option = {
    title: {
      text: "这次回答是否查阅了资料",
      left: "center"
    },
    tooltip: {
      trigger: "item",
      formatter: "{b}：{c}（{d}%）"
    },
    legend: {
      orient: "vertical",
      left: "left"
    },
    series: [
      {
        type: "pie",
        radius: "60%",
        data: [
          { name: "查阅了知识库资料", value: hasContext ? 1 : 0 },
          { name: "只靠模型记忆回答", value: hasContext ? 0 : 1 }
        ]
      }
    ]
  };

  return <ReactECharts option={option} style={{ height: 260 }} />;
};

