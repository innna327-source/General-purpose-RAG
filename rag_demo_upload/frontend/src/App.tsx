import React, { useState } from "react";
import { Layout, Typography, Input, Button, Space, message, Card, Tabs } from "antd";
import { askQuestion, type SimpleRAGResponse } from "./api/client";
import { ContextBarChart } from "./components/ContextBarChart";
import { ContextPieChart } from "./components/ContextPieChart";

const { Header, Content, Footer } = Layout;
const { Title, Paragraph, Text } = Typography;
const { TextArea } = Input;

const App: React.FC = () => {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<SimpleRAGResponse | null>(null);

  const onAsk = async () => {
    const q = question.trim();
    if (!q) {
      message.warning("请输入你想咨询的问题，例如：我最近总是头痛，应该怎么办？");
      return;
    }
    try {
      setLoading(true);
      const res = await askQuestion(q);
      setData(res);
    } catch (e: any) {
      console.error(e);
      message.error(e.message || "调用后端接口失败，请稍后重试。");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Header style={{ background: "#fff", borderBottom: "1px solid #eee" }}>
        <Title level={3} style={{ margin: 0 }}>
          RAG 问答与可视化面板
        </Title>
      </Header>

      <Content style={{ padding: 24 }}>
        <Space direction="vertical" size="large" style={{ width: "100%" }}>
          <Card>
            <Title level={4}>在这里输入你的问题</Title>
            <Paragraph type="secondary">
              尽量用完整、自然的中文来描述你的情况，例如：
              <Text code>我最近总是头痛，还睡不好，应该怎么办？</Text>
            </Paragraph>
            <TextArea
              rows={4}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="请输入你的问题……"
            />
            <div style={{ marginTop: 16, textAlign: "right" }}>
              <Button type="primary" loading={loading} onClick={onAsk}>
                发送给后端分析
              </Button>
            </div>
          </Card>

          <Card>
            <Title level={4}>分析结果</Title>
            {!data && <Paragraph>还没有结果，请先在上面输入问题并点击按钮。</Paragraph>}

            {data && (
              <Tabs
                items={[
                  {
                    key: "answer",
                    label: "文字说明（小白友好）",
                    children: (
                      <div>
                        <Paragraph strong>AI 给你的建议</Paragraph>
                        <Paragraph
                          style={{
                            whiteSpace: "pre-wrap",
                            background: "#fafafa",
                            padding: 12,
                            borderRadius: 4
                          }}
                        >
                          {data.answer}
                        </Paragraph>
                        <Paragraph type="secondary">{data.explanation}</Paragraph>
                      </div>
                    )
                  },
                  {
                    key: "chart",
                    label: "图表理解这次回答",
                    children: (
                      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                        <div style={{ flex: 1, minWidth: 280 }}>
                          <ContextBarChart items={data.related_items} />
                        </div>
                        <div style={{ flex: 1, minWidth: 280 }}>
                          <ContextPieChart items={data.related_items} />
                        </div>
                      </div>
                    )
                  },
                  {
                    key: "raw",
                    label: "原始数据（进阶用户可看）",
                    children: (
                      <pre
                        style={{
                          maxHeight: 360,
                          overflow: "auto",
                          background: "#111827",
                          color: "#e5e7eb",
                          padding: 12,
                          borderRadius: 4,
                          fontSize: 12
                        }}
                      >
                        {JSON.stringify(data, null, 2)}
                      </pre>
                    )
                  }
                ]}
              />
            )}
          </Card>
        </Space>
      </Content>

      <Footer style={{ textAlign: "center" }}>
        日志位置：项目根目录下的 <Text code>logs/service.log</Text>，每次前端请求和后端调用都会记录在这里。
      </Footer>
    </Layout>
  );
};

export default App;

