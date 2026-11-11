import {fireEvent,render,screen} from '@testing-library/react'
import {expect,it,vi} from 'vitest'
import {RewardLearningPanel} from './RewardLearningPanel'
import {captureRewardReference} from '../domain/rewardLearning'
import type {GraphModel} from '../domain/types'

it('keeps reward controls open across captures and saves while reflecting imported reward sources',()=>{
  const graph:GraphModel={learningRate:.01,nodes:[
    {id:'data',type:'dataset',label:'data',position:{x:0,y:0},params:{dataset:'class-scores'}},
    {id:'weight',type:'weight',label:'weight',position:{x:0,y:0},params:{value:1}},
    {id:'loss',type:'loss',label:'loss',position:{x:0,y:0},params:{loss:'cross-entropy'}},
  ],edges:[]}
  const props={disabled:false,onChange:vi.fn(),onRun:vi.fn()}
  const view=render(<RewardLearningPanel {...props} graph={graph} filename="qa.json"/>)
  fireEvent.click(screen.getByText('Reward learning',{exact:true}))
  expect(screen.getByText('Reward learning',{exact:true}).closest('details')).toHaveAttribute('open')
  const captured=captureRewardReference(graph,'preferences')
  view.rerender(<RewardLearningPanel {...props} graph={captured} filename="preferences.json"/>)
  expect(screen.getByText('Reward learning',{exact:true}).closest('details')).toHaveAttribute('open')
  expect(screen.getByLabelText('Reward source')).toHaveValue('preferences')
  fireEvent.change(screen.getByLabelText('Reward source'),{target:{value:'verifiable'}})
  expect(screen.getByLabelText('Reward source')).toHaveValue('verifiable')
  view.rerender(<RewardLearningPanel {...props} graph={captured} filename="imported-preferences.json"/>)
  expect(screen.getByLabelText('Reward source')).toHaveValue('preferences')
  expect(screen.getByText('Reward learning',{exact:true}).closest('details')).toHaveAttribute('open')
})
