import React from "react";
import ReactECharts from "echarts-for-react";
import type { SimpleContextItem } from "../api/client";

interface Props {
  items: SimpleContextItem[];
}

// 用非常直观的方式展示“参考了多少条资料”
export const ContextBarChart: React.FC<Props> = ({ items }) => {
  if (!items.length) {
    return <div>当前这次回答，只是根据模型自身能力生成，没有额外参考文档。</div>;
  }

  const option = {
    title: {
      text: "本次回答参考的资料数量",
      left: "center"
    },
    tooltip: {
      trigger: "axis",
      formatter: "{c} 条相关资料"
    },
    xAxis: {
      type: "category",
      data: ["本次回答"]
    },
    yAxis: {
      type: "value",
      name: "资料条数"
    },
    series: [
      {
        data: [items.length],
        type: "bar",
        label: {
          show: true,
          position: "top",
          formatter: "{c} 条"
        }
      }
    ]
  };

  return <ReactECharts option={option} style={{ height: 260 }} />;
};

