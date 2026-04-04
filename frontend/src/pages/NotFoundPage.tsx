import { Button, Result } from 'antd'
import { useNavigate } from 'react-router-dom'

export function NotFoundPage() {
  const navigate = useNavigate()

  return (
    <Result
      status="404"
      title="页面不存在"
      subTitle="当前地址没有对应页面，请返回数据集中心继续操作。"
      extra={
        <Button type="primary" onClick={() => navigate('/datasets')}>
          返回数据集中心
        </Button>
      }
    />
  )
}
